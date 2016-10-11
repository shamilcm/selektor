import logging
import argparse
import cPickle
from selektor import data

logger = logging.getLogger("selektor.train")


if __name__=="__main__":    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--trainset-path", dest="train_fname", help="train file")
    parser.add_argument("-dev", "--devset-path", dest="dev_fname", help="development file")
    parser.add_argument("-test", "--testset-name", required=False, dest="test_fname", help="test file")
    parser.add_argument("--extract-features", dest="extract_feat", action='store_true', help="Set this flag to extract COUNT features.")
    parser.add_argument("-emb", "--emb-path", dest="w2v_fname", help="path/name of pretrained word embeddings (Word2Vec binary). ")
    parser.add_argument("-o", "--output-path", dest="outfname", help="Name of the output pickle file")

    parser.add_argument("--ques_col", type=int, default=0, help="column of question, default is 0")
    parser.add_argument("--ans_col", type=int, default=1, help="column of answer, default is 1")
    parser.add_argument("--lab_col", type=int, default=2, help="column of label, default is 2")
    args = parser.parse_args()

    revs, vocab, max_l = data.build_data(args.train_fname, args.dev_fname, args.test_fname, args.ques_col, args.ans_col, args.lab_col, extract_feature=args.extract_feat)

    wordvecs = None
    if args.w2v_fname is not None: # use word embeddings for CNN
        logger.info("loading and processing pretrained word vectors")
        wordvecs = data.WordVecs(args.w2v_fname, vocab, binary=1, has_header=False)

    cPickle.dump([revs, wordvecs, max_l], open(args.outfname, "wb"))
    logger.info("dataset created!")
    logger.info("end logging")
